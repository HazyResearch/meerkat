import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from 'tailwindcss'
import type { UserConfig } from 'vite';

const config: UserConfig = {
	plugins: [sveltekit()],
	css: {
		postcss: {
			plugins: [tailwindcss],
		},
	}
};

export default config;
