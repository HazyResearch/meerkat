import{S as W,i as q,s as L,k as d,a as k,l as c,m as u,h as i,c as M,n as s,I as T,b as G,G as a,B as $,w as S,x as N,y as z,f as H,t as B,z as C}from"../../../chunks/index-d0966100.js";import{N as K,F as Q}from"../../../chunks/Footer-026e7ad2.js";/* empty css                                                            */import{m as D}from"../../../chunks/marked.esm-75dc6516.js";function Z(w){let e,n,m=D.parse(w[0])+"",b,o,f,x,l,p=D.parse(w[1])+"",y,t,h,_,g,I=D.parse(w[2])+"",P,v,R,E,F,O=D.parse(w[3])+"";return{c(){e=d("div"),n=d("div"),b=k(),o=d("iframe"),x=k(),l=d("div"),y=k(),t=d("iframe"),_=k(),g=d("div"),P=k(),v=d("iframe"),E=k(),F=d("div"),this.h()},l(A){e=c(A,"DIV",{class:!0});var r=u(e);n=c(r,"DIV",{class:!0});var U=u(n);U.forEach(i),b=M(r),o=c(r,"IFRAME",{class:!0,src:!0}),u(o).forEach(i),x=M(r),l=c(r,"DIV",{class:!0});var V=u(l);V.forEach(i),y=M(r),t=c(r,"IFRAME",{class:!0,src:!0}),u(t).forEach(i),_=M(r),g=c(r,"DIV",{class:!0});var J=u(g);J.forEach(i),P=M(r),v=c(r,"IFRAME",{class:!0,src:!0}),u(v).forEach(i),E=M(r),F=c(r,"DIV",{class:!0});var j=u(F);j.forEach(i),r.forEach(i),this.h()},h(){s(n,"class","prose"),s(o,"class","mt-4 aspect-video md:w-[512px] md:h-[280px] sm:w-[256px] sm:h-[140px]"),T(o.src,f="https://youtube.com/embed/a8FBT33QACQ")||s(o,"src",f),o.allowFullscreen="allowfullscreen",s(l,"class","prose"),s(t,"class","mt-4 aspect-video md:w-[512px] md:h-[280px] sm:w-[256px] sm:h-[140px]"),T(t.src,h="https://youtube.com/embed/3ItA70qoe-o")||s(t,"src",h),t.allowFullscreen="allowfullscreen",s(g,"class","prose"),s(v,"class","mt-4 aspect-video md:w-[512px] md:h-[280px] sm:w-[256px] sm:h-[140px]"),T(v.src,R="https://youtube.com/embed/4Kk_LZbNWNs")||s(v,"src",R),v.allowFullscreen="allowfullscreen",s(F,"class","prose"),s(e,"class","w-full flex flex-col items-center flex-wrap mt-28 mb-16")},m(A,r){G(A,e,r),a(e,n),n.innerHTML=m,a(e,b),a(e,o),a(e,x),a(e,l),l.innerHTML=p,a(e,y),a(e,t),a(e,_),a(e,g),g.innerHTML=I,a(e,P),a(e,v),a(e,E),a(e,F),F.innerHTML=O},p:$,i:$,o:$,d(A){A&&i(e)}}}function Y(w){return[`
# Interactive Data Frames and Meerkat: A Path to Foundation Models as a Reliable Software Abstraction
_The Meerkat Team_

## The Rise of Unstructured Data

Recent progress in machine learning shows that [foundation models](https://crfm.stanford.edu/assets/report.pdf)—large machine learning models trained on massive amounts of data—can perform a remarkably wide range of tasks with reasonable proficiency. These models can even be taught to perform entirely new tasks through *in-context learning* with a small number of examples, e.g. using text prompts with a large language model. Foundation models range from text-only models like [GPT-3](https://arxiv.org/pdf/2005.14165.pdf), to multi-modal models that involve training on images, text, audio and video data e.g. vision-language models like [CLIP](https://openai.com/research/clip).

Over the past year, we’ve been thinking about how foundation models will impact the workflow of *technical teams* spanning software engineering, data science, and machine learning. The lines are blurring between these roles—software engineers and data scientists must now contend day-to-day with how to instruct and evaluate model APIs, and integrate these APIs into their workflows.

All of these teams routinely interact with unstructured data types (e.g videos, images, free text, etc.). However, deriving insights from unstructured data requires significant time and human effort for gathering annotations and performing quality control. These investments are out of reach for most teams. 

Our bet is this: *FMs will lower the barrier to entry for working with unstructured data, and technical teams will increasingly interact with these models to build new tools, surface insights, and deploy software.*

- *Data Science Teams.* Organizations invest people-hours in order to extract insights from unstructured data. For example, one large hospital system calculated the fraction of preventable adverse events by hiring medical providers to scour clinical notes ([Bates et al., 2023](https://www.nejm.org/doi/full/10.1056/NEJMsa2206117)). FMs have the potential to bring this resource-intensive process within reach of more hospitals ([Agrawal et al., 2022](https://arxiv.org/pdf/2205.12689.pdf)).
- *Software Engineering Teams.* Until this year, only highly-resourced teams could build autocomplete features over unstructured text  (*e.g.* [Google Docs Smart Compose](https://support.google.com/docs/answer/9643962?hl=en#zippy=%2Csmart-compose-in-google-docs-slides-sheets-drawings), [Microsoft Word AutoCorrect](https://www.notion.so/Release-Blogpost-4df54aa98ee04dc3b04314db78dbc0a9)). Now, we see *much* smaller teams produce autocomplete features with FMs that are *much* more expressive (*e.g.* [Notion AI](https://www.notion.so/product/ai), [Lex](https://lex.page/)).
- *Machine Learning Teams.* Well-resourced organizations use large labeling teams to identify groups of unstructured data points where the model is making mistakes (*e.g.* [Tesla’s data engine](https://www.braincreators.com/insights/teslas-data-engine-and-what-we-should-all-learn-from-it)). But, last year we showed that foundation models can help machine learning teams identify systematic errors made by models, potentially reducing the labeling burden ([Eyuboglu et al. 2022](https://ai.stanford.edu/blog/domino/)).

## Building Interactive Data Frames for Unstructured Data

As unstructured data permeates the work of technical teams, it’s critical that they have the right toolbox for wrangling it. For structured data, teams swear by data frames, like those provided by Pandas and R. Back in 2021, [we started wondering](https://www.notion.so/64891aca2c584f1889eb0129bb747863): *why doesn’t something similar exist for unstructured data?*

One reason is that the reliable software abstractions (*e.g.* NumPy) that power traditional DataFrame operations fall flat when applied to unstructured data. A filter over a structured column (*e.g.* \`df[df[”age”] > 18]\`) can be implemented with one line of NumPy code, but there is no simple abstraction that could implement a semantic  filter over unstructured data (*e.g.* \`df[df[”image”].contains(”person”)]\`).

What if we viewed foundation models as a software abstraction that processes unstructured data? Much like NumPy is to Pandas, this software abstraction would power a data frame for unstructured data.

**The problem?** FMs are a terrible software abstraction, and we’re not the first to notice this ([Bommasani et al.](https://crfm.stanford.edu/2022/11/17/helm.html), [Narayan et al.](https://www.vldb.org/pvldb/vol16/p738-narayan.pdf)). FMs are hard to control (e.g. brittle to prompt wording ([Arora, et al.](https://arxiv.org/abs/2210.02441)), often produce undesired outputs (e.g. hallucinate knowledge, perpetuate social biases), and require careful evaluation. The process of using a traditional software abstraction (*i.e.* reading the documentation and writing code) is very different than the process of using FMs, which lack the predictability of a good software abstraction.

People have found success using FMs by carefully instructing and testing them. But this process is miserable when done in code alone:

- Say we’re trying to filter a dataset of paintings based on artistic style. A vision-language FM can produce scores to filter on, but how do we verify that the FM is correct if we can’t see and label the images?
- Imagine we’re trying to extract structured data from PDFs using a foundation model, but it’s not working as expected. How do we provide feedback to the model if we can’t annotate or highlight the PDFs?
- FMs can be used to identify slices of data where a different machine learning model is systematically making mistakes. But, how can we interpret the discovered slices if we can’t inspect the data inside them?

Code is a poor interface for inspecting and annotating data, which are critical for providing feedback to and iterating on FMs. ***Interactive systems*** could enable us to control FMs, turning them from unpredictable actors into usable software abstractions. 

## Meerkat: Towards Interactive Data Systems

We’re excited to introduce [Meerkat](https://github.com/hazyresearch/meerkat), a Python library that teams can use to interactively wrangle their unstructured data with foundation models.

Meerkat’s approach is based on two pillars:

1. **Heterogeneous data frames with extended API.** At the heart of Meerkat is a data frame that can store structured fields (e.g. numbers, strings, and dates) alongside complex objects (*e.g.* images, web pages, audio) and their tensor representations (*e.g.* embeddings, logits) in a single table. 
Meerkat's data frame API goes beyond structured data analysis libraries like Pandas by providing a set of FM-backed unstructured data operations. For example, we can search a data frame for images of people using the Meerkat code below:
    
    \`\`\`python
    import meerkat as mk
    mk.search(
        df, 
        query=mk.embed("A photo of a person", engine="clip"),
        by=mk.embed(df["image"], engine="clip")
    )
    \`\`\`
    
2. **Interactive framework.** Meerkat provides graphical user interfaces that allow users to verify the outputs of FM operations (*e.g.* search) and provide feedback via further instruction or labeling.
Critically, Meerkat GUIs are remarkably composable — users can build custom data applications all from within Python. These apps can also be used within Notebook environments, allowing users to transition easily between code and GUIs, and streamlining the process of instructing and evaluating FMs. 

Without Meerkat, you would use existing tools that either prioritize high-quality interactive components and force you to write your own data abstractions (e.g. Streamlit), or prioritize data abstractions with no support for interactive workflows (e.g. pandas), forcing you to write an interactive application from scratch. With Meerkat, you can use a *single* data frame to spin up interfaces, store data gathered from user interaction, and conduct analyses that require running FMs, regardless of data modality! 

Overall, we’re optimistic that these two pillars will simplify the process of working with foundation models and unstructured data, and make it it easier for technical teams to extract insights and meaning from complex and diverse datasets.

Let’s see how these principles look in practice with a few examples.


## Demos: Meerkat In Practice

We'll go through three demos, where each demo will be loosely oriented around the workflow of a different technical team.

### Demo 1 (Data Science): Analyzing a Dataset of Paintings in a Notebook

Identifying patterns in image data is challenging without visualization and interactivity. Here’s a demo where Meerkat makes the analysis of art data possible in a notebook alone.
`,`
*A demo of Meerkat on a dataset of art images in a Jupyter notebook. We spin up interactive interfaces powered by vision-language FMs to explore this data, and create an on-the-fly interface in Python that combines data exploration and plotting.*

### Demo 2 (Software Engineering): Using Spreadsheet-Style Flash Fill to Analyze PDFs

Software teams that want to build products around unstructured data will need to program and control these models, as well as evaluate and check if they work correctly on internal validation data. Here’s a demo where Meerkat simplifies this process by mixing code and user interaction.
`,`
*A demo of Meerkat where we take on the role of a software engineer working on a bibliography manager. We analyze a dataset of scientific papers, using the flash fill component to fill in missing metadata. This component was built using the Meerkat framework in < 300 lines of pure Python code.*

### Demo 3 (Machine Learning): Interactive Error Analysis of an Image Classifier

Machine learning teams routinely perform error analyses when building models, but often struggle to extract nuanced information about the errors. Here’s a demo that shows how Meerkat can be used to perform fine-grained error analyses. (Meerkat is already being used by this community, and we’re continuing to support this set of users as a finalist of the Stanford HAI AI Audit challenge.)
`,`
*A standalone Meerkat application for fine-grained error analysis of image classification models. Combining traditional dataframe and new FM-powered components can help us get a deeper understanding of the errors made by an image classifier.*

## Conclusion
Meerkat is new Python library that makes it easier to work with unstructured data and foundation models by bringing them into close proximity via interactive data frames.
We’re excited to see what the community can do with Meerkat. [Join our Discord](https://discord.gg/pw8E4Q26Tq) if you want to talk to us, use Meerkat or build on top of it, and definitely give Meerkat a try!

## Acknowledgments
Thanks to all the readers who provided feedback on this post and this release: Dean Stratakos, Geoff Angus, Shreya Rajpal, Dan Fu, Brandon Yang, Neel Guha, Sarah Hooper, Simran Arora, Mayee Chen, Michael Zhang, Khaled Saab, Kabir Goel, Gordon Downs, Joel Johnson and Zachary Marion.
`]}class X extends W{constructor(e){super(),q(this,e,Y,Z,L,{})}}function ee(w){let e,n,m,b,o,f,x,l,p,y;return m=new K({props:{page:"blog"}}),f=new X({}),p=new Q({}),{c(){e=d("div"),n=d("header"),S(m.$$.fragment),b=k(),o=d("main"),S(f.$$.fragment),x=k(),l=d("footer"),S(p.$$.fragment),this.h()},l(t){e=c(t,"DIV",{class:!0});var h=u(e);n=c(h,"HEADER",{class:!0});var _=u(n);N(m.$$.fragment,_),_.forEach(i),b=M(h),o=c(h,"MAIN",{class:!0});var g=u(o);N(f.$$.fragment,g),g.forEach(i),x=M(h),l=c(h,"FOOTER",{class:!0});var I=u(l);N(p.$$.fragment,I),I.forEach(i),h.forEach(i),this.h()},h(){s(n,"class","fixed w-full top-0 z-50"),s(o,"class","relative grow"),s(l,"class","sticky bottom-0 z-50"),s(e,"class","flex flex-col h-screen")},m(t,h){G(t,e,h),a(e,n),z(m,n,null),a(e,b),a(e,o),z(f,o,null),a(e,x),a(e,l),z(p,l,null),y=!0},p:$,i(t){y||(H(m.$$.fragment,t),H(f.$$.fragment,t),H(p.$$.fragment,t),y=!0)},o(t){B(m.$$.fragment,t),B(f.$$.fragment,t),B(p.$$.fragment,t),y=!1},d(t){t&&i(e),C(m),C(f),C(p)}}}class se extends W{constructor(e){super(),q(this,e,null,ee,L,{})}}export{se as default};
