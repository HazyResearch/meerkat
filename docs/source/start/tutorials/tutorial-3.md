---
file_format: mystnb
kernelspec:
  name: python3
---

# Tutorial 3: Running

In this tutorial, we will build an interface to ask questions to a large language model (LLM).

Through this tutorial, you will learn about:
- the concept of **endpoints**
- how to pass **endpoints** to components to run on events
- Using `Textbox` and `Button` components
- Advanced: Using [`manifest`](https://github.com/hazyresearch/manifest), a server-client manager for running LLMs

```{admonition} Prerequisites
:class: tip
This tutorial assumes you have already completed [Tutorial 1](tutorial-1.md) and [Tutorial 2](tutorial-2.md).
```

To get started, run the tutorial demo script

```{code-block} bash
mk demo tutorial-2
```

You should see the tutorial app when you open the link in your browser.

Let's break down the code in the demo script.

**