<script lang="ts">
	export let data: { [key: string]: number } = {};

	const formatNumber = (number: number) => {
		if (Number.isInteger(number) && number < 1000) {
			return number.toString();
		}
		const symbolToDivisor = [
			['B', 9],
			['M', 6],
			['K', 3]
		];
		let numberStr = number.toFixed(2);
		for (let i = 0; i < symbolToDivisor.length; i++) {
			const symbol: string = symbolToDivisor[i][0];
			const divisor: number = Math.pow(10, symbolToDivisor[i][1]);
			if (number >= divisor) {
				numberStr = `${(number / divisor).toFixed(2)}${symbol}`;
				break;
			}
		}
		return numberStr;
	};
</script>

<div class="m-2 flex flex-wrap justify-center gap-x-2 gap-y-2">
	{#each Object.entries(data) as [k, v]}
		<div class="bg-slate-100 rounded-md flex flex-col shadow-sm py-0.5">
			<div class="text-slate-400 text-md px-2 self-center">{k}</div>
			<div class="font-bold text-xl self-center">
				{formatNumber(v)}
			</div>
		</div>
	{/each}
</div>
