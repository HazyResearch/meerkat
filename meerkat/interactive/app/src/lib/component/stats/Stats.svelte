<script lang="ts">
	export let data: { [key: string]: number } = {};

	const format_number = (number: number) => {
		if (Number.isInteger(number) && number < 1000) {
			return number.toString();
		}
		const symbol_to_divisor = [
			['B', 9],
			['M', 6],
			['K', 3]
		];
		let number_str = number.toFixed(2);
		for (let i = 0; i < symbol_to_divisor.length; i++) {
			const symbol: string = symbol_to_divisor[i][0];
			const divisor: number = Math.pow(10, symbol_to_divisor[i][1]);
			if (number >= divisor) {
				number_str = `${(number / divisor).toFixed(2)}${symbol}`;
				break;
			}
		}
		return number_str;
	};
</script>

<div class="m-2 flex flex-wrap justify-center gap-x-2 gap-y-2">
	{#each Object.entries(data) as [k, v]}
		<div class="bg-slate-100 rounded-md flex flex-col shadow-sm">
			<div class="text-slate-400 px-3 py-1 self-center">{k}</div>
			<div class="font-bold text-2xl self-center ">
				{format_number(v)}
			</div>
		</div>
	{/each}
</div>
