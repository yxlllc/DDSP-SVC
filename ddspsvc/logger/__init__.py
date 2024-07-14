from loguru import logger

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress as _Progress
from rich.progress import (
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console(stderr=None)
def addLogger(path):
    logger.add(
        path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}",
        colorize=True,
    )


info = logger.info
error = logger.error
warning = logger.warning
warn = logger.warning
debug = logger.debug

def hps(hps):
    console.print(hps)

def Progress():
    return _Progress(
        TextColumn("[progress.description]{task.description}"),
        # TextColumn("[progress.description]W"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("[red]*Elapsed[/red]"),
        TimeElapsedColumn(),
        console=console,
    )