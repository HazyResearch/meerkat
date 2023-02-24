<script lang="ts">
	export let title: string;
	export let byline: string;
	export let description: string;
</script>

<div class="flex flex-col flex-wrap p-6 bg-white rounded-lg shadow-lg dark:bg-gray-800">
	<slot name="header" />
	<div>
		<h4 class="mb-1 text-lg font-semibold text-gray-800 dark:text-white">{title}</h4>
		<h4 class="mb-1 text-sm text-gray-400 dark:text-white">{byline}</h4>

        <slot name="description">
            <p class="text-sm text-gray-600 dark:text-gray-400">
                {description}
            </p>
        </slot>
	</div>
</div>
