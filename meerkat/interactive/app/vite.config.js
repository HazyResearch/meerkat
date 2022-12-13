import { sveltekit } from '@sveltejs/kit/vite';
import path from 'path';
console.log(path.resolve(__dirname, 'src/lib/main.js'));
/** @type {import('vite').UserConfig} */
const config = {
	plugins: [sveltekit()],

	resolve: {
		alias: {
			// these are the aliases and paths to them
			$shared: path.resolve('./src/lib/shared'),
			$layouts: path.resolve('./src/lib/layouts'),
			$utils: path.resolve('./src/lib/utils'),
			$styles: path.resolve('./src/lib/styles'),
			$stores: path.resolve('./src/lib/stores'),
			$network: path.resolve('./src/routes/network')
		}
	},

	// build: {
	// 	lib: {
			// entry: path.resolve(__dirname, 'src/lib/main.js'),
			// name: 'Meerkat',
			// fileName: 'meerkat'
		// },
		// rollupOptions: {
		// 	// make sure to externalize deps that shouldn't be bundled
		// 	// into your library
		// 	external: ['svelte'],
		// 	output: {
		// 		// Provide global variables to use in the UMD build
		// 		// for externalized deps
		// 		globals: {
		// 			svelte: 'Svelte'
		// 		}
		// 	},
		// }
	// }
};

export default config;