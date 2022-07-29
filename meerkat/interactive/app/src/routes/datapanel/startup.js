import { count } from './stores.js';
import { get } from 'svelte/store';

let store = 0; 

export async function POST({ request }) {
	count.set(9999);
	count.update((n) => n + request.body.data_id);
	store = 1000; 
	return {
		status: 200
	};
}
// let countValue = 0;
// count.subscribe((value) => (countValue = value));
// console.log(count);

/** @type {import('@sveltejs/kit').RequestHandler} */
export async function GET() {
	// // let countValue = 0;
	// // count.subscribe((value) => (countValue = value));
	// // console.log(count);

	return {
		status: 200,
		headers: {
			'access-control-allow-origin': '*'
		},
		body: store
	};
}
