"""Adapted from
https://github.com/srush/streambook/blob/main/streambook/cli.py."""
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class MeerkatHandler(FileSystemEventHandler):
    def __init__(
        self,
        abs_path: str,
    ):
        self.abs_path = abs_path
        self.last_modified = datetime.now()
        super().__init__()

    def on_modified(self, event):
        if datetime.now() - self.last_modified < timedelta(seconds=1):
            return
        else:
            self.last_modified = datetime.now()

        if event is None or event.src_path == self.abs_path:
            print(f"Rerunning {self.abs_path}...")


def main(
    path: Path,
    watch: bool,
):
    abs_path = path.absolute()
    abs_dir = path.parent.absolute()
    event_handler = MeerkatHandler(
        abs_path=str(abs_path),
    )
    observer = Observer()

    command = ["python"]
    command = command + [str(abs_path)]

    if watch:
        print("Watching directory for changes:")
        print("\t", f"{abs_dir}")

        print("Command:")
        print("\t", " ".join(command))

        subprocess.run(command, capture_output=True)

        event_handler.on_modified(None)
        observer.schedule(event_handler, path=str(abs_dir), recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    main(
        path=args.path,
        watch=args.watch,
    )
