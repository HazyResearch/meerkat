<script lang="ts">
	// Load in getContext from svelte (always)
	import { getContext } from 'svelte';
	// Load in the type Writable from svelte/store (always)
	import type { Writable } from 'svelte/store';

	// Running getContext('Interface') returns an object which contains useful functions
	// for interacting with the Python backend.
    // Each of these functions can be accessed by running $function_name
	const { get_schema, get_rows, edit } = getContext('Interface');

    // This is a prop for our component. It is a Writable store, which means that it can be
    // read from and written to.
    // *** To access the value of the store, use $store_name, so e.g. $data ***
	export let df: Writable;
    export let doc_column: Writable<string>;

    $: rows_promise = $get_rows($df.box_id, 0, null, null, [$doc_column]);


	// let highlight_pred = (p: number) => {
	// 	if (p == 1) {
	// 		return 'deepskyblue';
	// 	} else {
	// 		return 'transparent';
	// 	}
	// };

	// let update_color = (c: string) => {
	// 	if (c == 'pos') {
	// 		return 'deepskyblue';
	// 	} else if (c == 'neg') {
	// 		return 'indianred';
	// 	} else {
	// 		return 'orange';
	// 	}
	// };

	// let update_color_light = (c: string) => {
	// 	if (c == 'pos') {
	// 		return '#c2ecff';
	// 	} else if (c == 'neg') {
	// 		return '#ffc2c2';
	// 	} else {
	// 		return '#ffe0a6';
	// 	}
	// };

	// function update_select(d, c) {
	// 	console.log(d);

	// 	let selected = document.getElementById('s' + d);
	// 	selected.style.background = update_color_light(c);
	// 	selected.style.color = '#000';

	// 	['pos', 'neg', 'que'].forEach(function (e) {
	// 		if (c == e) {
	// 			selected.querySelector('.i' + e).style.background = update_color(e);
	// 			selected.querySelector('.i' + e).style.color = '#fff';
	// 		} else {
	// 			selected.querySelector('.i' + e).style.background = '';
	// 			selected.querySelector('.i' + e).style.color = update_color(e);
	// 		}
	// 	});
	// }
</script>

{#await rows_promise}
    Waiting...
{:then dfrows}
	{dfrows.rows}
    <!-- {#each dfrows.rows as document}
        <div id="document">
            <div class="paragraph">
                {#each paragraph as sentence}
                    <div id="s${e.id}" class="sentence" style="border-color: ${highlight_pred(e.prediction)};">
                        ${e.sentence.replace('.', '')}.
                        <div class="text_interactions">
                            <div class="selecting" style="border-color: ${highlight_pred(e.prediction)};">
                                <i class="fa fa-check ipos" on:click={() => update_select('${e.id}', 'pos')} />
                                <i class="fa fa-times ineg" on:click={() => update_select('${e.id}', 'neg')} />
                                <i class="fa fa-question ique" on:click={() => update_select('${e.id}', 'que')} />
                            </div>
                        </div>
                    </div>
                {/each}
            </div>
        </div>
    {/each} -->
    
{/await}


<style>
	/* * {
		font-family: 'IBM Plex Sans', sans-serif;
	} */

	#document {
		position: fixed;
		left: 0;
		right: 0;
		top: 0;
		bottom: 0;
		display: flex;
		justify-content: center;
		align-items: center;
	}

	/* #text_space {
		max-width: 50vw;
	} */

	.paragraph {
		margin-bottom: 15px;
	}

	.sentence {
		display: inline;
		margin-right: 3px;
		font-size: 16px;
		line-height: 25px;
		position: relative;
		padding: 0;
		background: none;
		color: #8c8c8c;
		border: dotted 1px transparent;
	}

	.text_interactions {
		display: inline-flex;
		position: relative;
		width: 0;
		height: 0;
		overflow: hidden;
	}

	.selecting {
		position: absolute;
		left: 0;
		top: 0;
		margin-top: -17px;
		display: flex;
		align-items: center;
		right: 0;
		top: 0;
		z-index: 999;
		width: 60px;
		z-index: 99;
		background: #eaeaea;
		border-right: dotted 1px transparent;
		border-top: dotted 1px transparent;
		border-bottom: dotted 1px transparent;
	}

	.selecting i {
		display: flex;
		justify-content: center;
		align-items: center;
		font-size: 14px;
		height: 20px;
		width: 20px;
		justify-content: center;
		align-items: center;
	}

	i::before {
		display: flex;
		justify-content: center;
		align-items: center;
	}

	.sentence:hover {
		color: #000;
		background: #eaeaea;
		/*    box-shadow: 0 0 1000px 1000px rgb(255, 255, 255, 0.9);*/
		z-index: 999;
	}

	.sentence:hover > .text_interactions {
		overflow: visible;
	}

	.selecting > i:hover {
		background: #d1d1d1;
	}

	.selecting > .fa-check {
		color: deepskyblue;
	}

	.selecting > .fa-times {
		color: indianred;
	}

	.selecting > .fa-question {
		color: orange;
	}
</style>
