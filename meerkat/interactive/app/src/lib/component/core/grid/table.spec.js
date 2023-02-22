import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Table from './Table.svelte';

describe('Table', () => {
	it('should render Table', async () => {
		const { container, component } = render(Table, {
			props: {
                df: {
                    refId: "mock"
                }
			},
		});
        // Check the table is rendered
        // TODO: Table needs to be tested by writing a wrapper Component
        // and then testing the wrapper component since it is implicitly
        // reliant on DynamicComponent
		// expect(await screen.findByText('hello')).toBeTruthy();
		expect(container).toMatchSnapshot();

        // // Check the table is rendered
        // const table = screen.getByRole('table');
        // expect(table).toBeInTheDocument();
	});
});