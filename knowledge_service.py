"""前台知识导入服务：校验文件、抓取网页、保存原文并增量写入 Qdrant。"""

# 导入哈希工具，为网页生成稳定且不冲突的本地文件名。
from hashlib import sha256
# 导入 IP 地址判断工具，用于阻止网页抓取访问本机和内网。
from ipaddress import ip_address
# 导入日志模块，记录用户导入来源和切片数量。
import logging
# 导入 DNS 查询工具，在发起请求前检查目标地址。
import socket
# 导入 Path，安全创建 knowledge 分类目录和保存原始文件。
from pathlib import Path
# 导入正则表达式，清理分类名和文件名中的危险字符。
import re
# 导入 URL 解析工具，校验协议、主机并生成来源名称。
from urllib.parse import urljoin, urlparse

# 导入 HTTP 客户端，用于限制超时、重定向和流式读取网页。
import httpx

# 导入上传与网页体积限制。
from config import MAX_UPLOAD_MB, MAX_WEB_PAGE_MB
# 复用命令行导入中的文件解析、切片和 Contextual Retrieval。
from ingest import SUPPORTED_EXTENSIONS, build_documents, extract_text
# 导入单来源覆盖写入函数。
from rag import upsert_source_documents

# 创建当前模块的日志记录器。
logger = logging.getLogger(__name__)
# 定位项目中的知识库根目录。
KNOWLEDGE_DIR = Path(__file__).with_name("knowledge")


def safe_name(value: str, fallback: str) -> str:
    """把用户输入转换成不能跳出 knowledge 目录的安全名称。"""
    # 仅保留中文、字母、数字、空格、下划线、连字符和点号。
    cleaned = re.sub(r"[^\w\- .\u4e00-\u9fff]", "_", value, flags=re.UNICODE).strip(" .")
    # 限制长度，防止 Windows 或 Linux 文件系统拒绝过长名称。
    return (cleaned or fallback)[:100]


def source_location(category: str, filename: str) -> tuple[Path, str, str]:
    """返回安全保存路径、Qdrant source 和规范化分类。"""
    # 分类为空时使用通用分类；不允许斜杠生成任意嵌套目录。
    safe_category = safe_name(category, "通用")
    # 文件名只取 basename，再执行字符白名单清洗。
    safe_filename = safe_name(Path(filename).name, "document.txt")
    # 计算并创建分类目录。
    category_dir = KNOWLEDGE_DIR / safe_category
    category_dir.mkdir(parents=True, exist_ok=True)
    # source 使用与命令行导入一致的 POSIX 相对路径。
    source = f"{safe_category}/{safe_filename}"
    return category_dir / safe_filename, source, safe_category


def import_bytes(content: bytes, filename: str, category: str, original_url: str | None = None) -> dict:
    """解析一个上传文件，保存原文，并覆盖写入该来源的向量切片。"""
    # 空文件没有可索引内容。
    if not content:
        raise ValueError("文件内容为空")
    # 在解析前限制字节数，避免超大 PDF 或 HTML 占满内存。
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"文件不能超过 {MAX_UPLOAD_MB} MB")
    # 只允许明确支持的扩展名。
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("只支持 .txt、.pdf、.html、.htm 文件")
    # 先解析并确认存在正文，失败文件不会写入 knowledge 目录。
    text = extract_text(content, extension)
    # 生成受控的保存位置及 metadata。
    destination, source, safe_category = source_location(category, filename)
    # 保存原始文件，让下一次 ingest.py 全量重建时仍能包含前台上传内容。
    destination.write_bytes(content)
    # 使用文件 stem 作为 Qdrant Dashboard 中可读的 Point 名称前缀。
    documents = build_documents(text, source, safe_category, destination.stem)
    # 网页来源额外写入原始 URL，使回答的引用可以追溯到真实页面。
    if original_url:
        for document in documents:
            document.metadata["url"] = original_url
    # 同一 source 覆盖，其他来源不受影响。
    count = upsert_source_documents(documents)
    logger.info("uploaded knowledge imported | source=%s | chars=%d | chunks=%d", source, len(text), count)
    # 返回前台需要的结果摘要，不回传完整原文。
    return {"source": source, "category": safe_category, "characters": len(text), "chunks": count}


def validate_public_url(url: str) -> None:
    """只允许指向公网 HTTP(S) 主机的 URL。"""
    # 解析 URL 并限制协议，禁止 file、ftp 等协议读取服务器文件。
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("请输入完整的 http:// 或 https:// 网页地址")
    # URL 中不允许携带用户名密码。
    if parsed.username or parsed.password:
        raise ValueError("网页地址不能包含用户名或密码")
    try:
        # 检查域名解析出的每个 IP，任一内网地址都拒绝。
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("无法解析网页域名") from exc
    for address in addresses:
        target = ip_address(address)
        if not target.is_global:
            raise ValueError("不允许抓取本机、内网或保留地址")


def fetch_web_page(url: str) -> tuple[bytes, str]:
    """安全抓取一个 HTML 页面，并返回内容和最终 URL。"""
    # 每次重定向都重新校验，避免公开 URL 跳转到内网。
    current_url = url.strip()
    headers = {"User-Agent": "Simple-RAG-Importer/1.0"}
    with httpx.Client(timeout=httpx.Timeout(20.0, connect=8.0), follow_redirects=False, headers=headers) as client:
        for _ in range(6):
            validate_public_url(current_url)
            with client.stream("GET", current_url) as response:
                # 最多接受五次常见 HTTP 重定向。
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("网页重定向缺少目标地址")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                # 网页入口只接受 HTML，避免把任意下载内容当网页保存。
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise ValueError("该地址返回的不是 HTML 网页")
                limit = MAX_WEB_PAGE_MB * 1024 * 1024
                parts: list[bytes] = []
                size = 0
                # 流式累计并随时执行体积限制。
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise ValueError(f"网页内容不能超过 {MAX_WEB_PAGE_MB} MB")
                    parts.append(chunk)
                return b"".join(parts), str(response.url)
    raise ValueError("网页重定向次数过多")


def import_url(url: str, category: str) -> dict:
    """抓取网页、生成稳定 HTML 文件名并写入知识库。"""
    # 获取最终页面，重定向后的 URL 用作真实来源依据。
    content, final_url = fetch_web_page(url)
    # 从域名和 URL 哈希生成稳定文件名；重复导入同一页面会覆盖旧切片。
    parsed = urlparse(final_url)
    host = safe_name(parsed.hostname or "web", "web")
    digest = sha256(final_url.encode("utf-8")).hexdigest()[:12]
    result = import_bytes(content, f"{host}-{digest}.html", category, original_url=final_url)
    # 额外保存原始网页 URL，供页面明确展示抓取目标。
    result["url"] = final_url
    return result
