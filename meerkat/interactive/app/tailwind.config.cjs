const colors = require('tailwindcss/colors')


/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{html,js,svelte,ts,py}',
    '../../../**/*.py',
    "./node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}",
  ],
  theme: {
    extend: {
    },

  },
  plugins: [
    require('@tailwindcss/typography'),
    require('flowbite/plugin'),
  ],
  darkMode: 'class',
}
