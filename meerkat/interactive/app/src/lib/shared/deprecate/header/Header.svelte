<script type="ts">
	import Tabs from './Tabs.svelte';
	import Tab from './Tab.svelte';
import { compute_slots } from 'svelte/internal';

</script>

<div>
	<Tabs>
        {#if $$slots.match}
		    <Tab label="Match" id="match">
                <slot name="match"></slot>
		    </Tab>
        {/if}
		{#if $$slots.filter}
			<Tab label="Filter" id="filter">
				<slot name="filter"></slot>
			</Tab>
		{/if}
		<Tab label="Info" id="info">
			second
		</Tab>
	</Tabs>
</div>