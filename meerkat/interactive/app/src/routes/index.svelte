<script lang="ts">
	import { get as get_url } from '$lib/utils/requests';

	let promise = get_url('http://34.83.137.148/user/krandiash/proxy/7860/');
</script>

<h1>Welcome to SvelteKit</h1>
<p>Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation</p>

{#await promise}
	waiting...
{:then result}
	{result}
{:catch error}
	<p style="color: red">{error.message}</p>
{/await}
