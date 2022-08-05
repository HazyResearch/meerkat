<script lang="ts">
	export let data: string;
	export let height: string = "100%";
</script>

<img class="rounded-sm" style:height={height} src={data} alt="A cell." />
