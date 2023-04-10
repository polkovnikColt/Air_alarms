import pandas as pd
from IPython.core.display import HTML
import re
from bs4 import BeautifulSoup


def remove_header_content(page_html, debug=False):
    html = BeautifulSoup(page_html)
    paragraphs = html.findAll("p")

    for line in paragraphs:
        line_text = str(line.text)

        if len(line_text) < 200:
            if debug:
                print("Removing line: " + line_text)
            line.decompose()
        else:
            break

    return str(html)


def remove_links_and_references(page_html, debug=False):
    html = BeautifulSoup(page_html)
    paragraphs = html.find_all(["div", "p"])

    found_hr = False
    found_references_section = False

    for line in paragraphs:
        line_text = str(line.text)

        if line.find("hr"):
            found_hr = True
            continue

        if line_text.startswith("[1]") and not found_hr:
            found_hr = False
            continue

        if line_text.startswith("[1]") and found_hr:
            found_references_section = True

        if found_references_section:
            if debug:
                print("Removing line: " + line_text)
            line.decompose()

    found_references_section = False
    for line in paragraphs:
        line_text = str(line.text)

        if line_text.startswith("[1]") and (line.find("a") or "http" in line_text):
            found_references_section = True

        if found_references_section:
            if debug:
                print("Removing line: " + line_text)
            line.decompose()

    page_html = str(html)

    pattern = "\[(\d+)\]"
    page_html = re.sub(pattern, "", page_html)

    pattern = "r'http(\S+.*\s)"
    page_html = re.sub(pattern, "", page_html)

    return page_html


def remove_images_and_links(page_html, debug=False):
    html = BeautifulSoup(page_html)

    images = html.findAll("img")
    for img in images:
        img.decompose()

    links = html.findAll("a")
    for a in links:
        a.decompose()

    return str(html)


def remove_repeated_stuff(page_html, debug=False):
    html = BeautifulSoup(page_html)
    paragraphs = html.findAll("p")

    prefixes = ["Note: ISW does not ", "Satellite image ", "Appendix "]

    for line in paragraphs:
        line_text = str(line.text)

        if line_text.startswith(tuple(prefixes)):
            if debug:
                print("Removing line: " + line_text)
            line.decompose()

    return str(html)


def preprocess_page_html(page_html, debug=False):
    page_html = remove_header_content(page_html, debug)
    page_html = remove_links_and_references(page_html, debug)
    page_html = remove_images_and_links(page_html, debug)
    page_html = remove_repeated_stuff(page_html, debug)

    html = BeautifulSoup(page_html)
    return html.text
