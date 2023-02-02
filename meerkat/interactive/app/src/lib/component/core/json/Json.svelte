<script lang="ts">
	import { JsonView } from '@zerodevx/svelte-json-view';

	/**
	 * The JSON object to display.
	 */
	export let body: object;

	/**
	 * The left padding to use for the JSON object.
	 */
	export let padding: number = 2;

	/**
	 * Tailwind CSS class to apply to the wrapper div.
	 */
	export let classes: string = '';
</script>

<div
	class="font-mono text-sm text-gray-800 bg-purple-50 rounded-lg p-2 {classes}"
	style="--jsonPaddingLeft: {padding}rem;"
>
	<JsonView json={body} />
</div>
