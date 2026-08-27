"""项目统一日志配置。"""

# 导入标准日志模块。
import logging

# 导入环境中配置的日志级别。
from config import LOG_LEVEL


# 定义日志初始化函数，供 API 和导入脚本共同调用。
def configure_logging() -> None:
    """使用统一时间、级别、模块和消息格式初始化日志。"""
    # 初始化根日志；已有处理器时 force=True 可确保格式保持一致。
    logging.basicConfig(
        # 把字符串日志级别转换成 logging 常量，非法值时使用 INFO。
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        # 每条日志显示时间、级别、模块名和正文。
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        # 使用年月日和时分秒格式，便于对照一次请求的各阶段。
        datefmt="%Y-%m-%d %H:%M:%S",
        # 覆盖 Uvicorn 已创建的根处理器，让项目日志格式稳定。
        force=True,
    )
