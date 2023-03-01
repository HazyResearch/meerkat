# Welcome to Meerkat 🔮

```{admonition} Under Construction
:class: warning
Meerkat's documentation is still under construction to make it as comprehensive as possible. If you have any questions that are not addressed by the docs, please reach out to us on [Discord](https://discord.gg/pw8E4Q26Tq).
```


Meerkat's goal is to help technical teams work with *unstructured data* like images, text, audio etc., especially when using foundation models. We want to make foundation models a reliable software abstraction that can underlie applications that require reasoning about unstructured data.

Read more about our motivation for building Meerkat in our [blog post](https://hazyresearch.github.io/blog/meerkat). Long story short: we realized that to use foundation models reliably, we needed something that would make interactively working with unstructured data and models simple. We built Meerkat so users can focus on their problems and data and not on writing complicated, messy code.

Through our Python library and npm package, we offer tools to help you work with unstructured data and foundation models across many contexts:
- a simple, full-stack framework to build interactive user interfaces purely in Python,
- a DataFrame that is designed for use with unstructured data and that underlies all interactivity in Meerkat,
- support for notebooks, standalone scripts and even full scale web applications through integration with [SvelteKit](https://kit.svelte.dev/),
- out-of-the-box components and workflows in Python for common use cases,
- careful support for extending Meerkat to your needs, such as integrating custom components with minimal overhead.

As a young project, Meerkat is under active development and is actively looking for contributors and collaborators. **Join our [Discord](https://discord.gg/pw8E4Q26Tq).**  We're excited to be hands on and to help you build with Meerkat.
<!-- 
**Philosophy.** Our design philosophy is to give technically minded users the opportunity to extend and tinker to customize Meerkat to your liking, while giving most users the ability to use Meerkat without worrying about the details. 
We hope to make implementation decisions that prioritize simplicity, productivity and ergonomics over technical pyrotechnics and bloat.
 -->
#### About Us
Meerkat is being built by Machine Learning PhD students in the [Hazy Research](https://hazyresearch.stanford.edu) lab at Stanford. We're excited to build for a future where models will make it easier for teams to sift and reason through large volumes of data effortlessly. We have varied research backgrounds and have done research that touches all parts of the machine learning process: we've created new model architectures, studied model robustness and evaluation, worked on applications ranging from audio generation to medical imaging.


<div style="display: flex; gap: 2rem; align-items: center;">
    <a href="https://hazyresearch.github.io/">
        <img
            src="https://hazyresearch.stanford.edu/hazy-logo.png"
            alt="Hazy Research Logo"
            style="max-height: 80px;"
        />
    </a>
    <a href="https://crfm.stanford.edu/">
        <img
            src="https://crfm.stanford.edu/static/img/header/crfm-rgb.png"
            alt="Stanford CRFM Logo"
            style="max-height: 100px;"
        />
    </a>
    <a href="https://hai.stanford.edu/">
        <img
            src="https://hai.stanford.edu/themes/hai/stanford_basic_hai/lockup.svg"
            alt="Stanford HAI Logo"
            style="max-height: 60px;"
        />
    </a>
</div>

#### Acknowledgements
We would like to particularly acknowledge the following open-source projects that have made Meerkat possible to implement: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), [Apache Arrow](https://arrow.apache.org/), [HuggingFace](https://huggingface.co/), [Scikit-Learn](https://scikit-learn.org/), [Pydantic](https://pydantic-docs.helpmanual.io/), [FastAPI](https://fastapi.tiangolo.com/), [Typer](https://typer.tiangolo.com/), [SvelteKit](https://kit.svelte.dev/), [Svelte](https://svelte.dev/), [TailwindCSS](https://tailwindcss.com/), and [Flowbite](https://flowbite.com/).

We also want to acknowledge the following projects, which have provided inspiration for many of our design decisions: [Gradio](https://gradio.app/), [Streamlit](https://streamlit.io/), [Pynecone.io](https://pynecone.io/), [Plotly Dash](https://plotly.com/dash/), and [Shiny](https://github.com/rstudio/py-shiny).
