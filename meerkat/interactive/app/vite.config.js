import { sveltekit } from '@sveltejs/kit/vite';

/** @type {import('vite').UserConfig} */
const config = {
	plugins: [sveltekit()],

	test: {
		globals: true,
		environment: 'jsdom',

		// Add @testing-library/jest-dom matchers & mocks of SvelteKit modules
		setupFiles: ['./src/mocks/setup.ts'],

		// Exclude files in c8
		coverage: {
			exclude: ['./src/mocks']
		},

		deps: {
			inline: [/msw/, "vega-embed", "vega-loader", "vega-schema-url-parser"]
		}
	},

	build: {
		rollupOptions: {
			// Externalize deps
			external: [
				// // Exclude everything inside `deprecate` folders
				// TODO: KG restore this, it was failiing because
				// components in the deprecate folder had Python
				// bindings that were seen.
				// new RegExp('./src/lib/component/deprecate/.*'),
				// new RegExp('./src/lib/shared/deprecate/.*'),
			],
		}
	}
};

export default config;
