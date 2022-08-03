<script context="module">
	/** @type {import('./__types/[slug]').Load} */
	export async function load({ url }) {
		return {
			props: {
				_api_url: url.searchParams.get('api')
			}
		};
	}
</script>

<script lang="ts">
	import { api_url } from "./stores";
    export let _api_url: string;
	$api_url = _api_url;
</script>
