import adapter from '@sveltejs/adapter-static';
import preprocess from 'svelte-preprocess';


/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://github.com/sveltejs/svelte-preprocess
	// for more information about preprocessors
	preprocess: [
		preprocess({
			postcss: true,
		}),
	],

	kit: {
		adapter: adapter(),
		prerender: {
			entries: ['*', '/[slug]'],
		}
	},

	package: {
		files: (filename) => {
			// Exclude the `wrappers` and `ComponentContext` from the build
			// Exclude everything inside `deprecate` folders
			// Exclude any __pycache__ folders in any subdirectory
			// Exclude all .py files
			if (filename.match(/wrappers\/.*/)
				|| filename.match(/ComponentContext.*/)
				|| filename.match(/deprecate\/.*/)
				|| filename.match(/__pycache__\/.*/)
				|| filename.match(/.*\.py/)
			) {
				return false;
			}
			return true;
		}
	}
};

export default config;
