<script lang="ts">
    export let data: object;
    export let classes: string = '';
    export let altText: string = "A Meerkat cell."

    const images, numSlices = data;

    let slice: number = 0;
    
    function handleScroll(event: WheelEvent) {
        slice += event.deltaY * 0.01;

        slice = Math.floor(slice);

        // Restrict scale
        slice = Math.min(Math.max(0, slice), numSlices-1);
    }
</script>


  
<div on:wheel={handleScroll}>
    <img class={classes} src={images[slice]} alt={altText} />
</div>

