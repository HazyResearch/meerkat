<script lang="ts">
    export let classes: string = '';
    // One of 'top', 'bottom', 'left', 'right'
	export let align: string = 'top';
    // Whether to keep the toolbar always active (i.e. pinned).
    export let pin: boolean = false;
	export let isToolbarActive: boolean = false;
    export let zIndex: number = 20;
    export let isAbsolutePosition = true;

    if (pin) {
        isToolbarActive = true;
    }

    let outerClasses: string = '';
    if (align === "top" || align === "bottom") {
        outerClasses = 'flex w-full items-center ';
    } else {
        outerClasses = 'flex-col h-full justify-center ';
    }
    if (isAbsolutePosition) {
        outerClasses += 'absolute ';
    }
    if (zIndex > 0) {
        outerClasses += `z-index-${zIndex} `;
    }
    console.log("classes", classes);
</script>

<div
	class={'toolbar ' + outerClasses + ` ${align}-1 align-top`}
	on:mouseenter={pin ? ()=>{} : () => (isToolbarActive = true)}
	on:mouseleave={pin ? ()=>{} : () => (isToolbarActive = false)}
    on:wheel|preventDefault
>
	<div class={"w-full flex justify-between items-top " + classes}>
		{#if isToolbarActive || pin}
        <slot></slot>
		{/if}
	</div>
</div>


<style>
	.toolbar {
		padding: 0.1rem;
		height: 10%;
        justify-content: flex-end;
	}
</style>
