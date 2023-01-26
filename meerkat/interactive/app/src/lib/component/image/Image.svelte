<script lang="ts">
	export let data: string;
	export let classes: string = '';
</script>

<img class={"h-full w-full" + classes} src={data} alt="A Meerkat cell." />
