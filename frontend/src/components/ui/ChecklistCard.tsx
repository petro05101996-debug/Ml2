import React from "react";
export default function ChecklistCard({ items }: { items: string[] }) { return <ul>{items.map((i)=><li key={i}>{i}</li>)}</ul>; }
