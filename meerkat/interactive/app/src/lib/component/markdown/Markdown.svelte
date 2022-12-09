<script lang="ts">
	import Markdown from '@magidoc/plugin-svelte-marked';

	export let data: string;
</script>

<Markdown source={data} />
