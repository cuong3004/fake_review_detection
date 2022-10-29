# -*- coding: utf-8 -*-
 
# Importing Scrapy Library
import scrapy
 
# Creating a new class to implement Spide
class AmazonReviewsSpider(scrapy.Spider):
     
    # Spider name
    name = 'amazon_reviews'
     
    # Domain names to scrape
    allowed_domains = ['https://www.amazon.com', 'www.amazon.com', 'amazon.com']
     
    # Base URL for the World Tech Toys Elite Mini Orion Spy Drone
    myBaseUrl = "https://www.amazon.com/RESPAWN-110-Racing-Style-Gaming-Chair/product-reviews/B076HTJRMZ/ref=cm_cr_getr_d_paging_btm_next_208?ie=UTF8&reviewerType=all_reviews&pageNumber="
    # start_urls=[]
    
    # Creating list of urls to be scraped by appending page number a the end of base url
    # for i in range(1,2):
    #     start_urls.append(myBaseUrl+str(i))

    def __init__(self, myBaseUrl='', **kwargs):
        self.myBaseUrl = myBaseUrl
        self.start_urls = [f'{myBaseUrl}']
        print("myBaseUrl", self.myBaseUrl)
    
    # Defining a Scrapy parser
    def parse(self, response):
        #Get the Review List
        data = response.css('#cm_cr-review_list')
        
        #Get the Name
        name = data.css('.a-profile-name')
        
        #Get the Review Title
        title = data.css('.review-title')
        
        # Get the Ratings
        star_rating = data.css('.review-rating')
        # print(str(star_rating) + " OKOKOKOKOKOOK")
        
        # Get the users Comments
        comments = data.css('.review-text')
        count = 0
        """
        print(data.css('.a-last').css('a'))
        print(data.css('.a-last').css('a').attrib["href"])
        print("OKOKOKOKO")
        try:
        
        """
        try:
            next_page = "https://www.amazon.com" + data.css('.a-last').css('a').attrib["href"]
        except:
            next_page = None
        # combining the results
        for review in star_rating:
            yield{'Name':''.join(name[count].xpath(".//text()").extract()),
            'Title':''.join(title[count].xpath(".//text()").extract()),
            'Rating': ''.join(review.xpath('.//text()').extract()),
            'Comment': ''.join(comments[count].xpath(".//text()").extract())
            }
            count=count+1

        if next_page is not None:
            yield scrapy.Request(next_page, callback=self.parse)