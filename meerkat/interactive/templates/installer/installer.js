import { create } from 'create-svelte';

await create("../{{ name }}", {
    name: '{{ name }}',
    template: 'skeleton',
    types: 'typescript',
    prettier: true,
    eslint: true,
    playwright: false,
    vitest: false
});
