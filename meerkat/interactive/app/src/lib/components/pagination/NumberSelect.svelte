<script lang="ts">
	import Select from 'svelte-select';

    export let selection: {value: any, label: any} = {value: undefined, label: undefined};
	export let options: Array<{value: any, label: number}>;
	export let hint_text: string = 'Text';

    function handleSelect(event: any) {
        selection = event.detail;
    };
	
</script>


<div class="themed flex w-60">
    <div
		class="inline-flex justify-center w-3/5 p-2.5 border border-solid text-sm rounded-l-lg dark:bg-gray-800 dark:text-gray-100"
	>
		{hint_text}
	</div>
    <Select 
        value={selection.value}
        items={options} 
        containerClasses="w-2/5"
        placeholder={hint_text}
        isClearable={false}
        showIndicator={true}
        on:select={handleSelect}
    />
</div>

<style>	
	/* 	
			CSS variables can be used to control theming.
			https://github.com/rob-balfre/svelte-select/blob/master/docs/theming_variables.md
	*/
	
	.themed {
        --tw-bg-opacity: 1;
        --background: theme(colors.gray.600);
		--border: 1px solid;
		--borderRadius: 0px;
        --borderHoverColor: theme(colors.gray.600);
        --borderFocusColor: theme(colors.violet.600);
        --itemColor: theme(colors.gray.600);
        --itemHoverColor: theme(colors.gray.600);
        --itemActiveBackground: theme(colors.gray.800);
        --listBackground: theme(colors.gray.400);
        --itemHoverBG: theme(colors.gray.300);
        --itemIsActiveBG: theme(colors.violet.600);
	}
</style>