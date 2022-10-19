<script lang="ts">
	import {marked} from 'marked';

	export let value: string;

</script>

<div>{@html marked.parse(value)}</div>
