<script lang="ts">
    export let score: number;
    export let description: string;
</script>

<div class="flex space-x-2 text-sm rounded-md border px-2 py-0.5 shadow-sm">
	<p class="min-w-max italic">{ description }</p>
	<div class="text-green-500">&rho;={ score.toFixed(2) }</div>
</div>
