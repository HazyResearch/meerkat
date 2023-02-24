<script lang="ts">
	import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';

	export let data: string;
	export let isUrl: boolean = true;


	GlobalWorkerOptions.workerSrc = '../../../../../node_modules/pdfjs-dist/build/pdf.worker.js';

	let canvasRef;

	// console.log(url)

	// const data = atob(
	// 	'JVBERi0xLjcKCjEgMCBvYmogICUgZW50cnkgcG9pbnQKPDwKICAvVHlwZSAvQ2F0YWxvZwog' +
	// 		'IC9QYWdlcyAyIDAgUgo+PgplbmRvYmoKCjIgMCBvYmoKPDwKICAvVHlwZSAvUGFnZXMKICAv' +
	// 		'TWVkaWFCb3ggWyAwIDAgMjAwIDIwMCBdCiAgL0NvdW50IDEKICAvS2lkcyBbIDMgMCBSIF0K' +
	// 		'Pj4KZW5kb2JqCgozIDAgb2JqCjw8CiAgL1R5cGUgL1BhZ2UKICAvUGFyZW50IDIgMCBSCiAg' +
	// 		'L1Jlc291cmNlcyA8PAogICAgL0ZvbnQgPDwKICAgICAgL0YxIDQgMCBSIAogICAgPj4KICA+' +
	// 		'PgogIC9Db250ZW50cyA1IDAgUgo+PgplbmRvYmoKCjQgMCBvYmoKPDwKICAvVHlwZSAvRm9u' +
	// 		'dAogIC9TdWJ0eXBlIC9UeXBlMQogIC9CYXNlRm9udCAvVGltZXMtUm9tYW4KPj4KZW5kb2Jq' +
	// 		'Cgo1IDAgb2JqICAlIHBhZ2UgY29udGVudAo8PAogIC9MZW5ndGggNDQKPj4Kc3RyZWFtCkJU' +
	// 		'CjcwIDUwIFRECi9GMSAxMiBUZgooSGVsbG8sIHdvcmxkISkgVGoKRVQKZW5kc3RyZWFtCmVu' +
	// 		'ZG9iagoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDEwIDAwMDAwIG4g' +
	// 		'CjAwMDAwMDAwNzkgMDAwMDAgbiAKMDAwMDAwMDE3MyAwMDAwMCBuIAowMDAwMDAwMzAxIDAw' +
	// 		'MDAwIG4gCjAwMDAwMDAzODAgMDAwMDAgbiAKdHJhaWxlcgo8PAogIC9TaXplIDYKICAvUm9v' +
	// 		'dCAxIDAgUgo+PgpzdGFydHhyZWYKNDkyCiUlRU9G'
	// );

	// // Loaded via <script> tag, create shortcut to access PDF.js exports.
	// const pdfjsLib = window['pdfjs-dist/build/pdf'];

	// // The workerSrc property shall be specified.
	// pdfjsLib.GlobalWorkerOptions.workerSrc = '//mozilla.github.io/pdf.js/build/pdf.worker.js';

	getDocument(data)
		.promise.then((doc) => doc.getPage(2))
		.then((page) => {
			const scale = 1.5;
			const viewport = page.getViewport({ scale });

			// Prepare canvas using PDF page dimensions
			var context = canvasRef.getContext('2d');
			canvasRef.height = viewport.height;
			canvasRef.width = viewport.width;

			// Render PDF page into canvas context
			var renderContext = {
				canvasContext: context,
				viewport: viewport
			};

			page.render(renderContext);
		});
</script>

<svelte:head>
	<script src="//mozilla.github.io/pdf.js/build/pdf.js"></script>
</svelte:head>

<div class="aspect-auto">
<canvas bind:this={canvasRef} />
</div>