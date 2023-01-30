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
			inline: [/msw/]
		}
	},

	build: {
		rollupOptions: {
			// Externalize deps
			external: [
				// Exclude the `wrappers` and `ComponentContext` from the build
				new RegExp('./src/lib/wrappers/.*'),
				new RegExp('./src/lib/ComponentContext.*'),
				// Exclude everything inside `deprecate` folders
				new RegExp('./src/lib/component/deprecate/.*'),
				new RegExp('./src/lib/shared/deprecate/.*'),
			],
		}
	}
};

export default config;
