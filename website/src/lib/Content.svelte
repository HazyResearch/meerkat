<script lang="ts">
  	import { base } from '$app/paths';
	import Card from './Card.svelte';

	let words = ['Unstructured Datasets', 'Interactive Apps', 'Data Frames']
	let word_idx = 0;
</script>

<!-- Top Content Part, should remain below the navbar -->
<section
	class="font-rubik bg-white dark:bg-gray-900 bg-gradient-to-br border-b "
>
	<div class="container px-6 mx-auto pt-24">
		<div class="md:grid md:grid-cols-[2fr_3fr] items-center justify-between md:h-[400px] gap-14">
			<div class="max-w-xl mb-8 md:mb-0">
				<h1 class="text-3xl font-bold text-gray-800 md:text-5xl dark:text-white">
					<!-- <div class="flex flex-col align-top"> -->
					<div class="animate-pulse" 
						on:animationiteration={() => {
							word_idx = (word_idx + 1) % words.length;
					}}>
						{words[word_idx]}
					</div>
					meet 
					<div class="italic text-violet-600">Foundation Models.</div>
					<!-- </div> -->
				</h1>
				<p class="mt-4 text-gray-600 dark:text-gray-400">
					Meerkat is an open-source Python library, designed for technical teams to interactively wrangle images, videos, text documents and more with foundation models.
				</p>
			</div>
			<div class = "p-14 h-full self-icenter">
				<iframe src="https://youtube.com/embed/HJv9ZvtisN0?modestbranding=1"          allowfullscreen="allowfullscreen"
 				class="aspect-video h-full rounded-md shadow-lg"></iframe>
			</div>
			<!-- Insert Demo Video here -->
		</div>
	</div>
</section>
 

<!-- Add a section to show code for installing the Meerkat package. -->
<section
	class="font-rubik  dark:bg-gray-900 bg-gradient-to-bl  border-b "
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="flex flex-col items-center">
			<h1 class="text-4xl text-gray-800 dark:text-white">Install Meerkat</h1>
			<pre class="shadow-md rounded-lg mt-4 py-4 px-8 text-white bg-slate-800 dark:text-gray-400"><code
					>$ pip install meerkat-ml</code
				></pre>
		</div>
	</div>
</section>

<!-- Add a section with panels that show the different things we can do in Meerkat 
1. Data Frames for Unstructured Data
2. Custom Foundation Model Endpoints
3. Reactive Workflows Triggered by Endpoints
4. Interactive Web Interfaces with Scripting
5. Interfaces in Notebooks
-->
<!-- <section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-br border-b "
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="mb-8 flex flex-col items-center">
			<h1 class="text-4xl text-gray-800 dark:text-white">What can you do with Meerkat?</h1>
		</div>
		<div class="grid grid-cols-2 grid-flow-row gap-4">
			<Card
				title="Dataframes for Unstructured Data"
				byline="Images, Text, Audio, Video, and more."
			>
				<svelte:fragment slot="description">
					<p class="text-sm text-gray-600 dark:text-gray-400">
						Load any unstructured data into a Meerkat dataframe and use a simple API to wrangle data. 
					</p>
					<pre class="text-sm text-gray-600 dark:text-gray-400 max-md:hidden">
<code>
<code>import meerkat as mk</code>
<code>df = mk.DataFrame(...)</code>
<code>df.map(...)</code></code>
					</pre>
					<p class="text-sm text-gray-600 dark:text-gray-400">
						We're building support for running foundation models on your data, right in the dataframe.
					</p>
				</svelte:fragment>
			</Card>

			<Card
				title="Foundation Model Endpoints"
				description="Connect your data to foundation models and build application endpoints with our high-level wrappers around Pydantic and FastAPI."
			/>
			<Card 
				title="Reactive Workflows"
				description="Program workflows triggered by changes in your data or through user interaction."
			/>
			<Card 
				title="Full Scale Web Apps"
				description="Web apps and dashboards on top of your data and models, powered by SvelteKit."
			/>
			<Card 
				title="Interactivity in Jupyter Notebooks"
				description="The same interfaces that you use in web apps run automatically in Jupyter Notebooks, or vice-versa."
			/>
			<Card
				title="Public Sharing"
			>
				<svelte:fragment slot="description">
					<p class="text-sm text-gray-600 dark:text-gray-400">
						Just add <code>shareable=True</code> to your interface and share it with anyone.
					</p>
				</svelte:fragment>
			</Card>
			<Card
				title="One-Click Deployment"
				byline="Coming Soon"
			>
				<svelte:fragment slot="description">
					<p class="text-sm text-gray-600 dark:text-gray-400">
						Deploy dashboards, internal tools or public-facing apps with a single click.
					</p>
				</svelte:fragment>
			</Card>
		</div>
	</div>
</section> -->


<!-- Add a section with cards for different user personas: Researchers, Data Scientists, Developers, etc. -->
<!-- <section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-bl from-orange-200 via-red-200 to-purple-200 border-b "
> -->
<section
	class="font-rubik  dark:bg-gray-900 bg-gradient-to-br  border-b "
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="mb-8 flex flex-col items-center">
			<h1 class="mb-4 text-4xl text-gray-800 dark:text-white">Built for technical teams</h1>
			<div class="md:grid md:grid-cols-3 justify-items-center gap-4 mt-2 flex flex-col">
				<Card 
					title="🤖️ Machine Learning Teams " 
					description="Graphical user interfaces to prompt and control foundation models, collect feedback and iterate, all with Python scripting." 
					byline=""
				/>
				<Card 
					title="🧪️ Data Science Teams " 
					description="Data frames, visualizations and interactive data analysis over unstructured data in Jupyter Notebooks with pure Python." 
					byline=""
				/>
				<Card 
					title="👨‍💻️ Software Engineering Teams " 
					description="Fully custom applications in SvelteKit that seamlessly connect to unstructured data and model APIs in Python."
					byline=""
				/>
			</div>
		</div>
	</div>	
</section>


<!-- Section on who built this -->
<!-- <section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-br  border-b "
> -->
<section class="font-rubik md:grid-cols-[1fr_2fr] grid xl:px-36">

	<div class="container px-6 py-16 mx-auto md:py-8 h-full">
		<div class="flex flex-col items-center h-full">
			<h1 class="text-3xl text-gray-800 dark:text-white">Built by...</h1>
			<div class="flex flex-wrap justify-center mt-6 bg-slate-50 h-full rounded-lg shadow-lg px-4 gap-0.5 py-3 items-center">
				<div class="h-16 mx-1  text-gray-800 dark:text-gray-400 dark:bg-gray-800">
					<a href="https://hazyresearch.github.io/" class="flex w-full h-4/5 hover:drop-shadow-sm">
						<img
							src="https://hazyresearch.stanford.edu/hazy-logo.png"
							alt="Hazy Research Logo"
							class="h-full rounded-md"
						/>
						<span class="ml-2 text-2xl font-bold text-red-800 align-middle self-center lg:whitespace-nowrap"
							>Hazy Research</span
						>
					</a>
				</div>
		

				<div class="h-16 mx-1  text-gray-800 dark:text-gray-400 dark:bg-gray-800">
					<a href="https://crfm.stanford.edu/">
						<img
							src="https://crfm.stanford.edu/static/img/header/crfm-rgb.png"
							alt="Stanford CRFM Logo"
							class="h-full hover:shadow-sm"
						/>
					</a>
				</div>
					<div class="h-16 -mx-1  text-gray-800 dark:text-gray-400 dark:bg-gray-800">
					<a href="https://www.james-zou.com/people" class="flex w-full h-full hover:drop-shadow-sm">
						<img
							src="https://identity.stanford.edu/wp-content/uploads/sites/3/2020/07/block-s-right.png"
							alt="Hazy Research Logo"
							class="h-full rounded-md"
						/>
						<div class="ml-2 text-2xl font-mono text-red-800 align-middle self-center flex flex-col">
							<span class="text-sm">Biomedical</span> 
							<span class="text-sm">Data Science</span>
						</div>
					</a>
				</div>
			</div>
		</div>
	</div>

<!-- <div
	</section>
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-bl from-orange-200 via-red-200 to-purple-200"
> -->
<!-- </section>
	class="font-rubik dark:bg-gray-900 bg-gradient-to-br  border-b "
> -->
	<div class="container px-6 py-16 mx-auto md:py-8 h-full">
		<div class="mb-8 flex flex-col items-center h-full">
			<h1 class="text-3xl text-gray-800 dark:text-white">...with the support of</h1>
			<div class="flex flex-wrap justify-center mt-6 p-4 shadow-lg rounded-lg bg-slate-50 gap-2 h-full">
				<div
					class="h-12 mx-2 my-2 text-gray-800  hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://hai.stanford.edu/">
						<img
							src="https://hai.stanford.edu/themes/hai/stanford_basic_hai/lockup.svg"
							alt="Stanford HAI Logo"
							class="h-full bg-blue-900 p-2 rounded-sm"
						/>
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://svelte.dev/">
						<img
							src="https://svelte.dev/svelte-logo-horizontal.svg"
							alt="Svelte Logo"
							class="h-full"
						/>
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://tailwindcss.com/">
						<img
							src="https://tailwindcss.com/_next/static/media/tailwindcss-mark.79614a5f61617ba49a0891494521226b.svg"
							alt="Tailwind Logo"
							class="h-full"
						/>
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://flowbite.com/">
						<img
							src="https://flowbite.com/images/logo.svg"
							alt="Flowbite Logo"
							class="h-full"
						/>
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://pydantic-docs.helpmanual.io/">
						<img
							src="https://docs.pydantic.dev/logo-white.svg"
							alt="Pydantic Logo"
							class="h-full filter-black"
						/>
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://fastapi.tiangolo.com/">
						<img
							src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
							alt="FastAPI Logo"
							class="h-full"
						/>
					</a>
				</div>
				<div
					class="h-12  mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://pandas.pydata.org/">
						<img
							src="https://pandas.pydata.org/static/img/pandas.svg"
							alt="Pandas Logo"
							class="h-full"
						/>
					</a>
				</div>
				<div
					class="h-12  mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://numpy.org/">
						<img src="https://numpy.org/images/logo.svg" alt="Numpy Logo" class="h-full" />
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://pytorch.org/">
						<img
							src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png"
							alt="Pytorch Logo"
							class="h-full"
						/>
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800  hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://arrow.apache.org/">
						<img
							src="https://arrow.apache.org/img/arrow-inverse.png"
							alt="Apache Arrow Logo"
							class="h-full filter-black"
						/>
					</a>
				</div>
				<div
					class="h-12 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://huggingface.co/">
						<img
							src="https://huggingface.co/front/assets/huggingface_logo.svg"
							alt="Huggingface Logo"
							class="h-full"
						/>
					</a>
				</div>
			</div>
		</div>
	</div>
</section>


<style>
	.filter-black {
		filter: invert(100%) sepia(99%) saturate(14%) hue-rotate(345deg) brightness(106%) contrast(100%);
	}

	@keyframes pulse {
		100% {
        	opacity: 0.0;
		}
		90% {
        	opacity: 0.0;
		}
    	30% {
        	opacity: 1.0;
		}
		0% {
			opacity: 0.0;
		}
	}
	.animate-pulse {
		animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}
</style>