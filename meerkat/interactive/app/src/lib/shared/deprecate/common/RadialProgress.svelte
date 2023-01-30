<script lang="ts">
	export let percent: number;
	let circumference = 30 * 2 * Math.PI;
    console.log(circumference - percent / 100 * circumference)
</script>

<svg class="w-20 h-20">
	<circle
		class="text-gray-300"
		stroke-width="5"
		stroke="currentColor"
		fill="transparent"
		r="30"
		cx="40"
		cy="40"
	/>
	<circle
		class="text-blue-600"
		stroke-width="5"
		:stroke-dasharray={circumference}
		:stroke-dashoffset={circumference - percent / 100 * circumference}
		stroke-linecap="round"
		stroke="currentColor"
		fill="transparent"
		r="30"
		cx="40"
		cy="40"
	/>
</svg>
