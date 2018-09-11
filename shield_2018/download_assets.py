"""download_assets.py - download shield-2018-assets

The download-assets command uses the Github API to download the
GIT LFS atlas assets. This is a work-around to Dockerhub not having GIT LFS
support.
"""
import argparse
import logging
import os
import requests
from xml.dom.minidom import parseString as parseXMLString

content_url = \
    "http://leviathan-chunglab.mit.edu/shield-2018/atlas/"


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
    #
    # The apache server serves the directory listing like this:
    # <html><body><ul><li><a href="Parent directory">...</a></li>
    #       <li><a href="filename">...</a></li>...</ul></body></html>
    #
    content = requests.get(content_url + "?F=0").content.decode("ascii")
    html_start = content.find("<html>")
    doc = parseXMLString(content[html_start:])
    body = doc.getElementsByTagName("body")[0]
    ul = body.getElementsByTagName("ul")[0]
    for entry in ul.getElementsByTagName("a")[1:]:
        name = entry.getAttribute("href")
        filename = os.path.join(args.destination, name)
        url = content_url + name
        logging.info("Writing %s" % filename)
        with open(filename, 'wb') as fd:
            for chunk in requests.get(url).iter_content(chunk_size=4096):
                fd.write(chunk)


if __name__ == "__main__":
    main()
