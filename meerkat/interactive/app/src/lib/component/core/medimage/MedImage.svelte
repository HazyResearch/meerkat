<script lang="ts">
    export let data: Array<string>;
    export let classes: string = '';
    export let numSlices: number = null;

    // TODO: this should be reactive.
    if (numSlices === null) {
        numSlices = data.length;
    }
    let sliceNumber: number = Math.floor(numSlices / 2);
    
    function handleScroll(event: WheelEvent) {
        if (numSlices === 1) {
            return;
        }
        sliceNumber += event.deltaY * 0.5;
        sliceNumber = Math.floor(sliceNumber);

        // Restrict slices to the range [0, numSlices-1].
        sliceNumber = Math.min(Math.max(0, sliceNumber), numSlices-1);
    }
</script>



<img on:wheel|preventDefault={handleScroll} class={classes} src={data[sliceNumber]} alt='A medical image.' />
