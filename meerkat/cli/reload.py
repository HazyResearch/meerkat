"""
Adapted from https://github.com/srush/streambook/blob/main/streambook/cli.py
and https://medium.com/@jnavarr56/use-file-watching-and-subprocesses-to-develop-python-scripts-with-live-reloading-9ffaa66fd648
"""
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from threading import Timer
from typing import List
import difflib

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import tempfile, shutil, os
def create_temporary_copy(path):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'temp_file_name')
    shutil.copy2(path, temp_path)
    return temp_path

class MeerkatHandler(FileSystemEventHandler):
    __proc = None
    __handler_func = None
    __temp_file = None
    __command = None
    
    def __init__(
        self,
        abs_path: str,
    ):
        self.abs_path = abs_path
        self.last_modified = datetime.now()
        super().__init__()
        
    # Command to run the passed in script.
    @staticmethod
    def run(command: List[str]):
        # Run the python script and keep track of the process.
        MeerkatHandler.__proc = subprocess.Popen(command)
        MeerkatHandler.__command = command
        
        # Make a temp file that copies the contents of the script over
        # so that we can watch for changes to the script.
        MeerkatHandler.__temp_file = create_temporary_copy(MeerkatHandler.__command[1])


    def on_modified(self, event):
        if datetime.now() - self.last_modified < timedelta(seconds=1):
            return
        else:
            self.last_modified = datetime.now()
            
        if not (event is None or event.src_path == self.abs_path):
            return
        
        # If there is previously running process
        # from a previous reload then kill it.
        if self.__proc != None:
            self.__proc.kill()
            
        # Debouncing:

        # If there is a script reload scheduled
        # to occur in the future, cancel it.
        if self.__handler_func != None:
            self.__handler_func.cancel()

        # Schedule the reload to happen in .5 secs.
        self.__handler_func = Timer(.5, partial(self.run, self.__command))
        self.__handler_func.start()


def main(
    path: Path,
    watch: bool,
):
    abs_path = path.absolute()
    abs_dir = path.parent.absolute()
    event_handler = MeerkatHandler(
        abs_path=str(abs_path),
    )

    command = ["python"]
    command = command + [str(abs_path)]
    
    # Run the script
    MeerkatHandler.run(command=command)

    observer = Observer()
    if watch:
        print("Watching directory for changes:")
        print("\t", f"{abs_dir}")

        print("Command:")
        print("\t", " ".join(command))

        # subprocess.run(command)#, capture_output=True)

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
