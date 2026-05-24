import React from "react";
export default function UnknownFieldCard({ title, why, check }: { title: string; why: string; check: string }) { return <div><h4>{title}</h4><p>{why}</p><p>{check}</p></div>; }
