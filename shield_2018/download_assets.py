"""download_assets.py - download shield-2018-assets

The download-assets command uses the Github API to download the
GIT LFS atlas assets. This is a work-around to Dockerhub not having GIT LFS
support.
"""
import argparse
import logging
import os
import requests

content_url = \
    "https://api.github.com/repos/chunglabmit/shield-2018-assets/contents/atlas"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination",
                        help="Download directory")
    parser.add_argument("--log-level",
                        default="INFO",
                        help="The log level for the Python logger: one of "
                        '"DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".')
    parser.add_argument("--log-file",
                        help="File to log to. Default is console.")
    parser.add_argument("--log-format",
                        help="Format for log messages. See "
                        "https://docs.python.org/3/howto/logging.html"
                        "#changing-the-format-of-displayed-messages for help")
    return parser.parse_args()


def main():
    # The content is a list of dictionaries.
    args = parse_args()
    logging_kwargs = {}
    if args.log_level is not None:
        logging_kwargs["level"] = getattr(logging, args.log_level.upper())
    if args.log_format is not None:
        logging_kwargs["format"] = args.log_format
    if args.log_file is not None:
        logging_kwargs["filename"] = args.log_filename
    logging.basicConfig(**logging_kwargs)

    if not os.path.isdir(args.destination):
        os.makedirs(args.destination)
    content = requests.get(content_url).json()
    for entry in content:
        # The "name" key is the filename.
        name = entry["name"]
        filename = os.path.join(args.destination, name)
        # The "download_url" key tells you how to get it
        url = "https://github.com/chunglabmit/shield-2018-assets/blob/master/atlas/%s?raw=true" % name
        logging.info("Writing %s" % filename)
        with open(filename, 'wb') as fd:
            for chunk in requests.get(url).iter_content(chunk_size=4096):
                fd.write(chunk)


if __name__ == "__main__":
    main()
