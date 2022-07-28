export async function GET({ params }) {


  // redirect to the newly created item
  return {
    status: 200,
    body: {
      id: params.id
    }
  };
}