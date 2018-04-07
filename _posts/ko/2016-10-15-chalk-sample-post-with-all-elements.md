---
layout: post
title: "모든 요소를 가진 Chalk 샘플 포스트"
description: "초크에서 사용할 수 있는 모든 미리 설계된 요소를 살펴보십시오."
thumb_image: "documentation/sample-image.jpg"
tags: [design, jekyll]
lang: ko
---

초크는 보석 루주를 강조하는 기본 지킬 구문을 사용합니다. 그것은 밝고 어두운 테마 모두에 대한 사용자 정의 스타일을 가지고 있습니다.
`highlight` 태그를 사용하여 원하는 언어를 강조 표시하는 다음 코드를 사용하십시오 :

{% highlight html %}
<!-- This is a comment with highlight tag -->
<div class="grid">
  <h1>This is a heading</h1>
  <p>
    This is a paragraph text.
  </p>
</div>
{% endhighlight %}

``` html
<!-- This is a comment with ``` -->
<div class="grid">
  <h1>This is a heading</h1>
  <p>
    This is a paragraph text.
  </p>
</div>
```

그리고`<div class = "grid">a</div>`를 강조하는 인라인 구문.

## 제목

초크는 기본적으로 3 개의 표제를 포함합니다 :

## 첫 번째 수준
### 제목 두 번째 수준
#### 제목 3 수준

{% highlight markdown %}
## 첫 번째 수준
### 제목 두 번째 수준
#### 제목 3 수준
{% endhighlight %}

## 목록

정렬되지 않은 목록 예 :
* 정렬되지 않은 목록 항목 1
* 정렬되지 않은 목록 항목 2
* 정렬되지 않은 목록 항목 3
* 정렬되지 않은 목록 항목 4

순서가 지정된 목록 예 :
1. 주문 된 목록 항목 1
2. 주문 된 목록 항목 1
3. 주문 된 목록 항목 1
4. 주문 된 목록 항목 1

{% highlight markdown %}
* Unordered list item 1
* Unordered list item 2

1. Order list item 1
2. Order list item 1
{% endhighlight %}

## 이모지 지원 :star:

이모티콘은 어디에서나 사용할 수 있습니다. :cat: :cat2: 귀하의 자격 상향 조정!

## 인용

인용은 이렇게 보여집니다:

> Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna.

{% highlight markdown %}
> Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna.
{% endhighlight %}

## 미디어

이미지는 기본 `<img>`태그로 추가 될 수 있습니다.
이미지를 클릭하여 확대 할 수 있기를 원하면 이미지 포함 태그를 사용하십시오. 3 개의 변수를 전달할 수 있습니다.
- `path` : 블로그 포스트에 보여줄 이미지.
- `path-detail` : 확대 할 때 보여줄 이미지.
- `alt` : 블로그 게시물의 이미지 대체 텍스트.

{% include image.html path="documentation/sample-image.jpg" path-detail="documentation/sample-image@2x.jpg" alt="Sample image" %}

{% highlight liquid %}
{% raw %}
{% include image.html path="documentation/sample-image.jpg"
                      path-detail="documentation/sample-image@2x.jpg"
                      alt="Sample image" %}
{% endraw %}
{% endhighlight %}

동영상을 추가하고 기본적으로 반응합니다 (기본 4x3, 추가 수업 16x9).

<div class="embed-responsive embed-responsive-16by9">
<iframe src="https://www.youtube.com/embed/vO7m8Hre72E?modestbranding=1&autohide=1&showinfo=0&controls=0" allowfullscreen></iframe>
</div>

{% highlight html %}
<div class="embed-responsive embed-responsive-16by9">
  <iframe src="url-to-video" allowfullscreen></iframe>
</div>
{% endhighlight %}
