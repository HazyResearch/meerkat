Introduction to Interactive GUIs
--------------------------------

Meerkat allows users to build interactive applications on top of Meerkat 
dataframes. 

One of the main reasons for this functionality is because working with unstructured data 
frequently involves playing with it. And there's no better way to do that than through
a simple, interactive application. These applications can range from simple forms to gather user input inside Jupyter notebooks,
to full-blown dashboards and web applications that are deployed to the cloud.

Underneath the hood, Meerkat relies on a FastAPI Python backend, coupled with 
Svelte for the frontend.

There are some key distinctions between Meerkat and other tools that allow users to build
interactive applications:

#. Compared to tools like `Streamlit <streamlit.io>`_, Meerkat is much more focused on supporting data and machine learning use cases. It is likely to be a better fit for users that want to bring the full power of a frontend framework like Svelte to their data science workflows.
#. Compared to tools like `Gradio <gradio.app>`_, Meerkat is much more focused on complex, reactive applications, rather than demos. It is likely to be a better fit for users that want to build full-blown applications that involve ML models, graphing and interactivity.
#. Compared to Python-to-React compilation approaches like `Pynecone <pynecone.io>`, Meerkat is more opinionated about providing a solution for the average data science and machine learning user.

Most data science and machine learning workflows revolve around Python, and Meerkat brings the 
ability to build reactive data apps in Python to these users.

In addition, Meerkat embraces making it as easy as possible
for users to write custom Svelte components if they desire. 
This can make it easy for users
with only a passing knowledge of Javascript/HTML/CSS to build custom components without 
having to deal with the intricacies of the web stack.
