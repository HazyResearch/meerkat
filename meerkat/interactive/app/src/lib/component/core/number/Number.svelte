<script lang="ts">
	export let data: any;
	export let dtype: string = "auto";
    export let precision: number = 3
    export let percentage: boolean = false
    export let classes: string = ""

    if(dtype === "auto"){
        if(typeof data === "number") {
            
            if(Number.isInteger(data)) {
                dtype = "int";
            } else {
                dtype = "float";
            }
        } else {
            dtype = "string"
        }
    }
    
    
    if (dtype === "float") {
        if (percentage) {
            data = (data * 100).toPrecision(precision) + "%";
        } else {
            data = data.toPrecision(precision);
        }
    }


</script>
<div class={classes}>
    {data}
</div>