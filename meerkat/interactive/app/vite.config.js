import { sveltekit } from '@sveltejs/kit/vite';
import path from 'path';

/** @type {import('vite').UserConfig} */
const config = {
	plugins: [sveltekit()],
	
	resolve: {
		alias: {
			// these are the aliases and paths to them
			$components: path.resolve('./src/lib/components'),
			$layout: path.resolve('./src/lib/components/layout'),
			$layouts: path.resolve('./src/lib/layouts'),
			$utils: path.resolve('./src/lib/utils'),
			$styles: path.resolve('./src/lib/styles'),
			$stores: path.resolve('./src/lib/stores'),
			$network: path.resolve('./src/routes/network')
		}
	}
	
};

export default config;
