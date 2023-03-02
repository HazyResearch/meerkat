import fnmatch
import os
import re


def file_find_replace(directory, find, replace, pattern):
    for path, _, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = re.sub(find, replace, s)
            with open(filepath, "w") as f:
                f.write(s)


if __name__ == "__main__":
    # Redirect the docs navbar Meerkat logo to the home page
    # of the website.
    file_find_replace(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "static", "docs"),
        r'<a class="navbar-brand text-wrap" href="#">',
        r'<a class="navbar-brand text-wrap" href="/">',
        "*html",
    )
