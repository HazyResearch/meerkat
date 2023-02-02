<script lang="ts">
	import { marked } from 'marked';

	/** The Markdown source to render. */
	export let body: string;

	/** The @tailwindcss/typography classes to apply to the rendered Markdown. */
	export let classes: string = "";

	/**
	 * The options to pass to the marked.js parser.
	 * @see https://marked.js.org/using_advanced#options
	 */

	/** The base URL to use for relative links. */
	export let baseUrl: string = null;

	/** If true, add <br> on a single line break (copies GitHub behavior on comments, but not on rendered markdown files). Requires gfm be true. */
	export let breaks: boolean = false;

	/** If true, use approved GitHub Flavored Markdown (GFM) specification. */
	export let gfm: boolean = true;

	/** If true, include an id attribute when emitting headings (h1, h2, h3, etc). */
	export let headerIds: boolean = true;

	/** A string to prefix the id attribute when emitting headings (h1, h2, h3, etc). */
	export let headerPrefix: string = '';

	/** A string to prefix the className in a <code> block. Useful for syntax highlighting. */
	export let langPrefix: string = 'language-';

	/** If true, autolinked email address is escaped with HTML character references. */
	export let mangle: boolean = true;

	/** If true, conform to the original markdown.pl as much as possible. Don't fix original markdown bugs or behavior. Turns off and overrides gfm. */
	export let pedantic: boolean = false;

	/** If true, sanitize the HTML passed into markdownString with the sanitizer function. */
	export let sanitize: boolean = false;

	/** If true, the parser does not throw any exception. */
	export let silent: boolean = false;

	/** If true, use "smart" typographic punctuation for things like quotes and dashes. */
	export let smartypants: boolean = false;

	/** If true, emit self-closing HTML tags for void elements (<br/>, <img/>, etc.) with a "/" as required by XHTML. */
	export let xhtml: boolean = false;

	// Set the options for the marked.js parser.
	marked.setOptions({
		baseUrl,
		breaks,
		gfm,
		headerIds,
		headerPrefix,
		langPrefix,
		mangle,
		pedantic,
		sanitize,
		silent,
		smartypants,
		xhtml
	});
</script>

<div class="prose {classes}">
	{@html marked.parse(body)}
</div>
