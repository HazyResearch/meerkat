# FAQs


```{dropdown} *I have been waiting for over a minute, and my interface is not loading*
Try changing the API port. By default the API port is `5000`, consider making it `500X` (e.g. `5005`).å
If you are launching the interface in a notebook, use `mk.gui.start(api_port=5005)`.
If you are running a script, pass argument `--api-port XXXX`.
```