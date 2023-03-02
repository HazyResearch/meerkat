import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/kit/vite';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    kit: {
        adapter: adapter(),
        prerender: {
			entries: ['*', '/[slug]'],
		}
    },
    preprocess: vitePreprocess()
};

export default config;
