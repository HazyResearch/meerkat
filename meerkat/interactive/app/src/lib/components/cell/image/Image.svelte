<script lang="ts">
	export let data: string;
	export let height: string = "100%";
</script>

<img class="rounded-md shadow-lg" style:height={height} src={data} alt="A cell." />
