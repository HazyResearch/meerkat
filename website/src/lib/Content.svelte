<script lang="ts">
	import Card from './Card.svelte';
</script>

<!-- Top Content Part, should remain below the navbar -->
<section
	class="font-rubik bg-white dark:bg-gray-900 bg-gradient-to-br from-purple-200 via-red-200 to-orange-200 border-b border-white"
>
	<div class="container px-6 mx-auto py-32">
		<div class="flex flex-col items-center justify-between md:flex-row">
			<div class="max-w-xl mb-8 md:mb-0">
				<h1 class="text-3xl font-bold text-gray-800 md:text-5xl dark:text-white">
					Foundation Models meet Data and Apps.
				</h1>
				<p class="mt-4 text-gray-600 dark:text-gray-400">
					Meerkat is an open-source Python library to wrangle unstructured and structured data using dataframes,
					connect data to foundation models, and build full scale web apps on top (or just run
					interfaces in a notebook).
				</p>
			</div>
			<!-- Insert Demo Video here -->
		</div>
	</div>
</section>


<!-- Add a section to show code for installing the Meerkat package. -->
<section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-bl from-orange-200 via-red-200 to-purple-200 border-b border-white"
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="flex flex-col items-center">
			<h1 class="text-4xl text-gray-800 dark:text-white">Install Meerkat</h1>
			<pre class="rounded-lg mt-4 py-4 px-8 text-gray-600 bg-white dark:text-gray-400"><code
					>pip install meerkat-ml</code
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
<section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-br from-purple-200 via-red-200 to-orange-200 border-b border-white"
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
</section>


<!-- Add a section with cards for different user personas: Researchers, Data Scientists, Developers, etc. -->
<section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-bl from-orange-200 via-red-200 to-purple-200 border-b border-white"
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="mb-8 flex flex-col items-center">
			<h1 class="mb-2 text-4xl text-gray-800 dark:text-white">For Everyone</h1>
			<!-- <p class="text-gray-600 dark:text-gray-400">
				Whether you're a researcher, data scientist, or developer, Meerkat is for you.
			</p> -->
			<!-- <div class="flex flex-row items-stretch flex-wrap mt-2"> -->
			<div class="grid grid-cols-3 justify-items-center gap-4 mt-2">
				<Card 
					title="Foundation Model Researchers 👨‍🔬️" 
					description="Create interfaces to interact with models, collect usage data and analyze it, all in Python." 
					byline="Python + Dataframes"
				/>
				<Card 
					title="AI Auditors 🕵" 
					description="Create auditing workflows that can be used by non-technical users, including for data exploration and labeling." 
					byline="Python"
				/>
				<Card 
					title="Data Scientists 🔬" 
					description="Create data visualizations and interfaces for exploratory data analysis over unstructured data, which run directly in Jupyter Notebooks, and on dashboards." 
					byline="Python + Jupyter"
				/>
				<Card 
					title="Full-Stack Developers 👨‍💻️" 
					description="Build fully custom applications in SvelteKit that seamlessly connect to unstructured data and model APIs in Python." 
					byline="Python + SvelteKit"
				/>
				<Card 
					title="Hackers 🤓" 
					description="Build quickly without the constraints of a rigid framework: Meerkat will integrate with any Python library, or Svelte or Javascript package with no effort." 
					byline="Python + SvelteKit"
				/>
				<Card 
				title="Front-End Developers 👨‍💻️" 
				description="Build and share (coming soon) beautiful components in Svelte for anyone to use in Python, and take advantage of our JavaScript API to interact with unstructured data backends easily." 
				byline="SvelteKit"
			/>
			</div>
		</div>
	</div>	
</section>



<!-- Use Cases -->
<section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-bl from-orange-200 via-red-200 to-purple-200 border-b border-white"
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="mb-8 flex flex-col items-center">
			<h1 class="mb-2 text-4xl text-gray-800 dark:text-white">Use Cases</h1>
			<div class="grid grid-cols-3 justify-items-center gap-4 mt-2">
				<Card 
					title="Model Demos" 
					description="" 
					byline="From simple to complex"
				/>
				<Card 
					title="Exploratory Data Analysis" 
					description="" 
					byline="Python"
				/>
				<Card 
					title="AI Auditing" 
					description="" 
					byline="Python + Jupyter"
				/>
				<Card 
					title="Internal Tools" 
					description="Build fully custom applications in SvelteKit that seamlessly connect to unstructured data and model APIs in Python." 
					byline="Python + SvelteKit"
				/>
				<Card 
					title="Data Visualizations" 
					description="Build quickly without the constraints of a rigid framework: Meerkat will integrate with any Python library, or Svelte or Javascript package with no effort." 
					byline="Python + SvelteKit"
				/>
				<Card 
				title="" 
				description="Build and share (coming soon) beautiful components in Svelte for anyone to use in Python, and take advantage of our JavaScript API to interact with unstructured data backends easily." 
				byline="SvelteKit"
			/>
			</div>
		</div>
	</div>	
</section>



<!-- Section on who built this -->
<section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-br from-purple-200 via-red-200 to-orange-200  border-b border-white"
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="mb-8 flex flex-col items-center">
			<h1 class="text-4xl text-gray-800 dark:text-white">Built by</h1>
			<div class="flex flex-wrap justify-center mt-6 bg-gray-100 rounded-lg shadow-lg">
				<div class="h-20 p-4 mx-1 my-1 text-gray-800 dark:text-gray-400 dark:bg-gray-800">
					<a href="https://hazyresearch.github.io/" class="flex w-full h-full hover:shadow-sm">
						<img
							src="https://hazyresearch.stanford.edu/hazy-logo.png"
							alt="Hazy Research Logo"
							class="rounded-md"
						/>
						<span class="ml-2 text-2xl font-bold text-red-800 align-middle self-center"
							>Hazy Research</span
						>
					</a>
				</div>
				<div class="h-20 p-2 mx-1 my-1 text-gray-800 dark:text-gray-400 dark:bg-gray-800">
					<a href="https://crfm.stanford.edu/">
						<img
							src="https://crfm.stanford.edu/static/img/header/crfm-rgb.png"
							alt="Stanford CRFM Logo"
							class="w-full h-full hover:shadow-sm"
						/>
					</a>
				</div>
			</div>
		</div>
	</div>
</section>

<section
	class="font-rubik bg-purple-50 dark:bg-gray-900 bg-gradient-to-bl from-orange-200 via-red-200 to-purple-200"
>
	<div class="container px-6 py-16 mx-auto md:py-8">
		<div class="mb-8 flex flex-col items-center">
			<h1 class="text-4xl text-gray-800 dark:text-white">We wouldn't be here without</h1>
			<div class="flex flex-wrap justify-center mt-6 bg-red-200 shadow-lg">
				<div
					class="h-20 p-2 mx-2 my-2 text-gray-800 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://cs.stanford.edu">
						<img
							src="SUSig_2color_Stree_StnfrdOnly_Left.png"
							alt="Stanford University Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-5 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://hai.stanford.edu/">
						<img
							src="https://hai.stanford.edu/themes/hai/stanford_basic_hai/lockup.svg"
							alt="Stanford HAI Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://svelte.dev/">
						<img
							src="https://svelte.dev/svelte-logo-horizontal.svg"
							alt="Svelte Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-6 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://tailwindcss.com/">
						<img
							src="https://tailwindcss.com/_next/static/media/tailwindcss-mark.79614a5f61617ba49a0891494521226b.svg"
							alt="Tailwind Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://flowbite.com/">
						<img
							src="https://flowbite.com/images/logo.svg"
							alt="Flowbite Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://pydantic-docs.helpmanual.io/">
						<img
							src="https://docs.pydantic.dev/logo-white.svg"
							alt="Pydantic Logo"
							class="w-full h-full filter-black"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://fastapi.tiangolo.com/">
						<img
							src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
							alt="FastAPI Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://pandas.pydata.org/">
						<img
							src="https://pandas.pydata.org/static/img/pandas_white.svg"
							alt="Pandas Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://numpy.org/">
						<img src="https://numpy.org/images/logo.svg" alt="Numpy Logo" class="w-full h-full" />
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://pytorch.org/">
						<img
							src="https://pytorch.org/assets/images/pytorch-logo.png"
							alt="Pytorch Logo"
							class="w-full h-full"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://arrow.apache.org/">
						<img
							src="https://arrow.apache.org/img/arrow-inverse.png"
							alt="Apache Arrow Logo"
							class="w-full h-full filter-black"
						/>
					</a>
				</div>
				<div
					class="h-20 p-4 mx-2 my-2 text-gray-800 bg-red-200 hover:shadow-sm dark:text-gray-400 dark:bg-gray-800"
				>
					<a href="https://huggingface.co/">
						<img
							src="https://huggingface.co/front/assets/huggingface_logo.svg"
							alt="Huggingface Logo"
							class="w-full h-full"
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
</style>
