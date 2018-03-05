from logging import (getLogger, StreamHandler, FileHandler, CRITICAL, ERROR,
                     WARNING, INFO, DEBUG)


def get_logger(name, level=INFO, output_file=None):
    """
    Args:
        naame: getLoggerに渡す変数.
        level: ログ出力レベル. デフォルトはINFO.
        output_file: ログ出力ファイル. Noneの場合は標準出力. デフォルトはNone.
    """
    levels = [CRITICAL, ERROR, WARNING, INFO, DEBUG]
    if level not in levels:
        raise ValueError('"level" should be {}. Received "{}" insted.'.format(
            levels, level))

    logger = getLogger(name)

    handler = StreamHandler() if output_file is None else FileHandler(
        output_file, 'a+', encoding='utf-8')
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
