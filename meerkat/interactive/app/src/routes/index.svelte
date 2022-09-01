<script lang="ts">
	import { get as get_url } from '$lib/utils/requests';
	import { browser } from '$app/env';

	//let promise = get(`${api_url}/`)
	console.log('Browser: ', browser);

	let fn = async () => {
		return 'SSR';
	};

	let promise = fn();

    promise.then(
        (response) => console.log(response)
    )
</script>

{#await promise}
	waiting...
{:then result}
	{result}
{:catch error}
	<p style="color: red">{error.message}</p>
{/await}
